"""Exercise stage dates, two-node placement and immutable input binding."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import fvcom_lake as lake
from fvcom_lake_attempt import materialize
from fvcom_run_control import init_project

def main():
    with TemporaryDirectory() as tmp:
        root=Path(tmp);project=root/"study";request=root/"request.json";binding=root/"build.json"
        request.write_text(json.dumps(dict(schema="simulation_request_v1",case_id="lake",mode=lake.MODE,profile=lake.PROFILE,
            execution={"site":"perlmutter"},perlmutter={"account":"test","project_dir":"/scratch/lake","source_dir":"/scratch/lake/source"})))
        init_project(request,project)
        spec=json.loads((project/"stage_plan.json").read_text())
        assert [s["ranks"] for s in spec["stages"] if s["name"].startswith("benchmark_")]==[64,128,256]*3
        binding.write_text(json.dumps(dict(status="frozen",site="perlmutter",source_commit=lake.SOURCE_COMMIT,
            native_executable="/scratch/lake/source/fvcom",executable_sha256="a"*64,build_environment={"modules":["cpu/1"]})))
        files=("STARTUP_FILE","GRID_FILE","DEPTH_FILE","SIGMA_LEVELS_FILE","WIND_FILE","HEATING_FILE",
               "AIRPRESSURE_FILE","PRECIPITATION_FILE","STATION_FILE","RIVER_INFO_FILE")
        nml=root/"base.nml"
        nml.write_text("\n".join(("INPUT_DIR='./input/',","OUTPUT_DIR='./',","STARTUP_TYPE='hotstart',",
             "START_DATE='2025-07-01 00:00:00',","END_DATE='2025-07-02 00:00:00',",
             *(f"{k}='{k.lower()}.dat'," for k in files))))
        inp=root/"inputs.json";inp.write_text(json.dumps({k.lower()+".dat":"b"*64 for k in files}))
        args=argparse.Namespace(project=project,inputs=inp,binding=binding,name="benchmark_256_r1",
            namelist=nml,revision=0,walltime="01:00:00",qos="regular",case_name="superior")
        spec=materialize(args)
        folder=project/"run/native_lsofs/attempts"/spec["name"]
        job=(folder/"job.sbatch").read_text()
        assert "#SBATCH --ntasks-per-node=128" in job and "#SBATCH --nodes=2" in job
        assert materialize(args)==spec
        nml.write_text(nml.read_text().replace("2025-07-02","2025-07-03"))
        try: materialize(args)
        except ValueError: pass
        else: raise AssertionError("wrong science clock accepted")
    print("Lake profile, schedule, two-node launch and clock rejection passed.")
if __name__=="__main__": main()
