"""Offline regressions for freshwater physics and stage clocks in the native namelist."""
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import lake_configuration as lake
import fvcom_namelist as n


class LakeConfiguration(unittest.TestCase):
    def setUp(self):
        self.args=Namespace(start="2025-07-01T00:00:00Z",end="2025-08-01T00:00:00Z",
                            input_dir="../input",output_dir=".",extstep=2.5,isplit=2,startup="initial.nc",
                            surface="atmosphere.nc",pe="pe.nc",river_info="rivers.nml",river_number=20,
                            station="stations.dat",station_location="node")

    def test_physics_and_native_template(self):
        v=lake.lake_values(self.args)
        self.assertEqual(v["NML_PHYSICS"]["SEA_WATER_DENSITY_FUNCTION"],"'dens1'")
        self.assertEqual(v["NML_GRID_COORDINATES"]["GRID_FILE_UNITS"],"'degrees'")
        self.assertEqual(v["NML_PHYSICS"]["TEMPERATURE_ACTIVE"],"T")
        self.assertEqual(v["NML_PHYSICS"]["BAROTROPIC"],"F")
        self.assertEqual(v["NML_OPEN_BOUNDARY_CONTROL"]["OBC_ON"],"F")
        self.assertEqual(v["NML_SURFACE_FORCING"]["PRECIPITATION_FILE"],"'pe.nc'")
        self.assertEqual(v["NML_RIVER_TYPE"]["RIVER_INFLOW_LOCATION"],"'edge'")
        self.assertEqual(v["NML_STATION_TIMESERIES"]["OUT_INTERVAL"],"'seconds=3600.0'")
        self.assertEqual(v["NML_NETCDF"]["NC_OUT_INTERVAL"],"'seconds=21600.0'")
        self.assertEqual(v["NML_RESTART"]["RST_FIRST_OUT"],"'2025-07-02 00:00:00'")
        template="".join("&"+g+"\n"+"".join(" "+k+" = 0,\n" for k in d)+"/\n" for g,d in v.items())
        self.assertEqual(n.parse_values(n.render_template(template,v)),v)
        with self.assertRaisesRegex(ValueError,"required key missing"):
            n.render_template(template.replace(" AIRPRESSURE_ON = 0,",""),v)

    def test_time_and_bound_input_names(self):
        self.args.extstep=2.7
        with self.assertRaisesRegex(ValueError,"divide"):
            lake.lake_values(self.args)

        self.args.extstep=2.5
        self.args.startup="../external.nc"
        with self.assertRaisesRegex(ValueError,"basename"):
            lake.lake_values(self.args)

    def test_subdaily_continuation_restart(self):
        self.args.end="2025-07-01T01:00:00Z"
        self.assertEqual(lake.lake_values(self.args)["NML_RESTART"]["RST_FIRST_OUT"],"'2025-07-01 01:00:00'")

    def test_freshwater_coupled_heat_with_prescribed_stress(self):
        self.args.coare_bulk="superior_bulk.nc"
        self.args.pe="raw_rain.nc"
        v=lake.lake_values(self.args)
        sf=v["NML_SURFACE_FORCING"]
        self.assertEqual((sf["HEATING_ON"],sf["WIND_TYPE"],sf["PRECIPITATION_FILE"]),("F","'stress'","'raw_rain.nc'"))
        self.assertEqual(sf["WIND_FILE"],sf["AIRPRESSURE_FILE"])
        self.assertEqual(v["NML_PHYSICS"]["SCALAR_POSITIVITY_CONTROL"],"F")
        bulk=v["NML_HEATING_CALCULATED"]
        self.assertEqual((bulk["HEATING_CALCULATE_ON"],bulk["HEATING_FRESHWATER"],bulk["COARE_VERSION"]),
                         ("T","T","'COARE26Z'"))
        self.assertEqual(bulk["HEATING_CALCULATE_FILE"],"'superior_bulk.nc'")
        self.assertEqual([bulk[k] for k in ("ZUU","ZTT","ZQQ")],["10.0","2.0","2.0"])
        self.assertEqual(v["NML_NETCDF"]["NC_OUT_INTERVAL"], "'seconds=21600.0'")
        self.args.end="2025-07-01T00:10:00Z"
        self.assertEqual(lake.lake_values(self.args)["NML_NETCDF"]["NC_OUT_INTERVAL"], "'seconds=600.0'")
        self.args.coare_bulk=None
        self.assertEqual(lake.lake_values(self.args)["NML_NETCDF"]["NC_OUT_INTERVAL"], "'seconds=21600.0'")
        self.args.coare_bulk="../raw/superior_bulk.nc"
        with self.assertRaisesRegex(ValueError,"basename"):
            lake.lake_values(self.args)

    def test_bulk_raw_rain_and_coverage(self):
        import netCDF4 as nc
        a=self.args
        a.coare_bulk="bulk.nc"
        a.start="2025-07-01T00:10:00Z"
        a.end="2025-07-01T00:55:00Z"
        with TemporaryDirectory() as tmp:
            a.input_root=Path(tmp)
            for name in ("superior_cor.dat","lsofs_cor.dat"):
                (a.input_root/name).write_bytes(b"native latitudes")
            for name in (a.surface,a.coare_bulk,a.pe):
                with nc.Dataset(a.input_root/name,"w") as d:
                    for dim,count in (("time",2),("node",3),("nele",1)):
                        d.createDimension(dim,count)
                    for key,dim,vals in (("node_id","node",[1,2,3]),("element_id","nele",[1])):
                        d.createVariable(key,"i4",(dim,))[:]=vals
                    day=d.createVariable("Itime","i4",("time",))
                    ms=d.createVariable("Itime2","i4",("time",))
                    day.units="days since 2025-07-01 00:00:00";day[:]=0
                    ms.units="msec since 00:00:00";ms[:]=[0,3600000]
                    if name==a.coare_bulk:
                        for key,dim,unit in (("U10","nele","m s-1"),("V10","nele","m s-1"),
                            ("air_temperature","node","Celsius"),("relative_humidity","node","percent"),
                            ("air_pressure","node","Pa"),("short_wave","node","W m-2"),("long_wave","node","W m-2")):
                            var=d.createVariable(key,"f4",("time",dim));var.units=unit;var[:]=1
                    if name==a.pe:
                        for key in ("precip","evap"):
                            d.createVariable(key,"f4",("time","node"))[:]=0
            lake.check_coare_inputs(a)
            with nc.Dataset(a.input_root/a.pe,"a") as d:
                d["evap"][1,1]=-1e-7
            with self.assertRaisesRegex(ValueError,"zero prescribed vapor"):
                lake.check_coare_inputs(a)
            a.end="2025-07-02T00:00:00Z"
            with self.assertRaisesRegex(ValueError,"cover the integration window"):
                lake.check_coare_inputs(a)

    def test_native_checkpoint_origins_and_limiter(self):
        import netCDF4 as nc
        start = "2025-07-02T01:00:00Z"
        for origin in ("2018-01-01", "1858-11-17"):
            with nc.Dataset("clock", "w", diskless=True) as ds:
                ds.createDimension("time", 1)
                day = ds.createVariable("Itime", "i4", ("time",))
                millis = ds.createVariable("Itime2", "i4", ("time",))
                day.units = "days since " + origin + " 00:00:00"
                millis.units = "msec since 00:00:00"
                day[:] = (n.parse_utc(start, "start") - n.parse_utc(origin, "origin")).days
                millis[:] = 3600000
                lake.check_startup_clock(ds, start)
                with self.assertRaisesRegex(ValueError, "no startup record"):
                    lake.check_startup_clock(ds, "2025-07-02T02:00:00Z")
        self.args.scalar_positivity_control = False
        self.assertEqual(lake.lake_values(self.args)["NML_PHYSICS"]["SCALAR_POSITIVITY_CONTROL"], "F")


if __name__=="__main__":
    unittest.main()
