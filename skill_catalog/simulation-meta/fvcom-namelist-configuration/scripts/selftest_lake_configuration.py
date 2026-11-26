"""Offline regressions for freshwater physics and stage clocks in the native namelist."""
from argparse import Namespace
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
