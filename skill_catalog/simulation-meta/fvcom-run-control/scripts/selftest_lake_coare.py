"""Exercise public COARE reader/cold/warm launch freezes and rejection of mixed evidence."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import fvcom_lake_coare as c
import fvcom_lake as lake
from fvcom_lake_attempt import materialize
from fvcom_run_control import build_parser
from fvcom_coare_output import ledgers


class Coupled(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.pin=c.read(c.build_assets()/"source_pin.json")
        self.base=self.put("binding.json",dict(status="frozen",site="perlmutter",source_commit=lake.SOURCE_COMMIT,
            executable_sha256="1"*64,native_executable="/scratch/lake/source/fvcom",build_environment={"modules":["cpu"]}))
        self.derivative=dict(schema="fvcom_coare_diagnostic_build_v1",status="frozen",scope="diagnostic",
            source_commit=lake.SOURCE_COMMIT,changed=self.pin["changed"],cpp=["FRESHWATER_LAKE","HEATING_CALCULATED",
            "LAKE_COARE_DIAG","MULTIPROCESSOR","SPHERICAL","AIR_PRESSURE","WET_DRY"],sha256="2"*64,
            exe="/scratch/lake/diagnostic/fvcom",baseline_executable_sha256="1"*64,baseline_binding_sha256=c.digest(self.base))
        self.build=self.put("build.json",self.derivative)
        self.requirements=dict(input_dir="./input/",output_dir="./",startup_type="hotstart",grid_file="grid.dat",
            depth_file="dep.dat",sigma_levels_file="sig.dat",startup_file="start.nc",station_file="station.dat",
            river_info_file="river.nml",wind_file="surface.nc",airpressure_file="surface.nc",precipitation_file="rain.nc",
            heating_calculate_file="bulk.nc",coriolis_file="cor.dat",wind_on="T",wind_type="stress",wind_kind="variable",
            airpressure_on="T",airpressure_kind="variable",precipitation_on="T",precipitation_kind="variable",heating_on="F",
            heating_calculate_on="T",heating_freshwater="T",coare_version="COARE26Z",heating_calculate_type="flux",
            heating_calculate_kind="variable",scalar_positivity_control="F",zuu=10,ztt=2,zqq=2,iramp=0,
            extstep_seconds=2.5,isplit=2,start_date="2025-06-17",end_date="2025-06-19",nc_first_out="2025-06-17",
            nc_out_interval="seconds=21600",rst_first_out="2025-06-18",rst_out_interval="days=1",out_interval="seconds=3600")
        files={v:"a"*64 for k,v in self.requirements.items() if k.endswith("_file")}
        files["superior_cor.dat"]="a"*64
        self.inputs=self.put("inputs.json",files)
        self.forcing=self.put("forcing.json",dict(schema="fvcom_lake_hrrr_coare26z_diagnostic_v1",status="prepared_for_native_qualification",
            model_start_utc="2025-06-17",model_end_utc="2025-06-19",
            files={k:files[k] for k in ("superior_cor.dat","bulk.nc","rain.nc")},
            hydrodynamic_reference=dict(sha256=files["surface.nc"],used_fields=["uwind_stress","vwind_stress","air_pressure"])))
        errors=dict.fromkeys(["coare_wind_speed","short_wave","latent_mass_Wm2","heat_addition_Wm2","raw_rain",
            "rain_and_dew","evap","hydro_mslp","uwind_stress","vwind_stress"],0.)
        self.reader=self.put("reader.json",dict(scope="diagnostic",status="verified",executable_sha256="2"*64,
            history_sha256="c"*64,records={"June":dict(max_abs_error=errors)}))
        self.put("request.json",dict(schema="simulation_request_v1",case_id="lake",mode=lake.MODE,profile=lake.PROFILE,
            execution={"site":"perlmutter"},perlmutter={"account":"test","project_dir":"/scratch/lake","source_dir":"/scratch/lake/source"}))

    def put(self,name,obj):
        p=self.root/name;p.write_text(json.dumps(obj));return p

    def run_attempt(self,label):
        nml=self.root/"superior_run.nml"
        def fmt(v):
            return repr(v) if isinstance(v,str) and v not in ("T","F") else str(v)
        nml.write_text("\n".join(f"{k.upper()}={fmt(v)}," for k,v in self.requirements.items()))
        args=build_parser().parse_args(["lake-attempt","--name",label,"--project",str(self.root),"--inputs",str(self.inputs),
            "--namelist",str(nml),"--binding",str(self.base),"--walltime","00:30:00","--coare-build",str(self.build),
            "--coare-reader",str(self.reader),"--coare-forcing",str(self.forcing)])
        return materialize(args)

    def test_freeze_and_repeat(self):
        rec=self.run_attempt("coare_warm")
        self.assertEqual(rec,self.run_attempt("coare_warm"))
        out=self.root/"run/native_lsofs/attempts"/rec["name"]
        script=(out/"job.sbatch").read_text()
        self.assertIn(rec["case_sha256"],script)
        self.assertIn(c.digest(out/"runtime_checksums.sha256"),script)
        self.assertIn("/scratch/lake/diagnostic/fvcom",script)
        self.assertEqual(rec["executable_sha256"],"2"*64)
        self.assertEqual(rec["schema"],"fvcom_coare_diagnostic_attempt_v1")
        self.requirements["end_date"]="2025-06-18"
        self.assertEqual(self.run_attempt("coare_cold")["mpi_ranks"],128)
        self.requirements.update(end_date="2025-06-17T00:10:00",rst_first_out="2025-06-17T00:10:00",nc_out_interval="seconds=600")
        self.assertNotIn("coare_reader.json",self.run_attempt("coare_reader")["qualification"])

    def test_reject_monthly_build_forcing_and_reader_mismatches(self):
        with self.assertRaises(ValueError):self.run_attempt("production")
        self.requirements["end_date"]="2025-07-17"
        with self.assertRaises(ValueError):self.run_attempt("coare_warm")
        self.requirements["end_date"]="2025-06-19"
        self.build.write_text(json.dumps({**self.derivative,"changed":{}}))
        with self.assertRaises(ValueError):self.run_attempt("coare_warm")
        self.put("build.json",self.derivative)
        self.put("reader.json",{**c.read(self.reader),"executable_sha256":"9"*64})
        with self.assertRaises(ValueError):self.run_attempt("coare_cold")
        files=c.read(self.inputs)
        with self.assertRaises(ValueError):c.checked_sources({"../unsafe":files["bulk.nc"]},Path("/scratch/lake"))
        with self.assertRaises(ValueError):ledgers(self.root,128,24,0)

if __name__=="__main__":
    unittest.main()
