"""Offline tests of the hourly forcing physics, grid frame, and paired file contract."""

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import netCDF4 as nc
import numpy as np
from pyproj import CRS

from acquire_wind_pressure import validate_clock
from prepare_wind_pressure import (
    hour_axis,
    model_basis,
    obc_ids,
    psimple_drag,
    record,
    stamp,
    verify_bundle,
    write,
    write_pressure_slice,
)
from surface_fluxes_core import _time_variables, sha256_file
from surface_fluxes_validation import _wind_diagnostics_to_earth


class WindPressureTests(unittest.TestCase):
    def test_duplicate_and_missing_hours(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            p = root / "sample.nc"
            begin = datetime(2020, 1, 1, tzinfo=timezone.utc)
            end = begin.replace(hour=1)
            with nc.Dataset(p, "w") as ds:
                ds.createDimension("time", 2)
                var = ds.createVariable("time", "i4", ("time",))
                var.units = "hours since 2020-01-01 00:00:00"
                var[:] = [0, 1]
            validate_clock([{"path": p.name}], root, begin, end)
            with nc.Dataset(p, "r+") as ds:
                ds["time"][:] = [0, 0]
            with self.assertRaisesRegex(ValueError, "every requested hour"):
                validate_clock([{"path": p.name}], root, begin, end)

    def test_native_drag_thresholds_and_direction(self):
        speed = np.array([0, 10.99, 11.0, 25.0, 30.0])
        tx, ty = psimple_drag(-0.8 * speed, 0.6 * speed)
        cd = np.array([0.0012, 0.0012, 0.001205, 0.002115, 0.002115])
        np.testing.assert_allclose(tx, -0.8 * 1.2 * cd * speed**2)
        np.testing.assert_allclose(ty, 0.6 * 1.2 * cd * speed**2)

    def test_frame(self):
        lon = np.array([-124.0, -123.0, -122.0])
        lat = np.full(3, 46.0)
        c, s = model_basis(CRS.from_epsg(32610), lon, lat)
        self.assertLess(s[0], 0)
        self.assertGreater(s[2], 0)
        np.testing.assert_allclose(c * c + s * s, 1)
        ux, vy = 4 * c - 3 * s, 4 * s + 3 * c
        np.testing.assert_allclose(c * ux + s * vy, 4)
        np.testing.assert_allclose(-s * ux + c * vy, 3)

    def test_native_slice_and_clock(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            grd, obc = root / "mesh.dat", root / "obc.dat"
            grd.write_text(
                "Node Number = 3\nCell Number = 1\n1 1 2 3\n1 10 20 4\n2 11 20 4\n3 10 21 5\n"
            )
            obc.write_text("OBC Node Number = 2\n1 3 1\n2 1 1\n")
            ids = obc_ids(obc, 3)
            np.testing.assert_equal(ids, [3, 1])
            clock = hour_axis("2020-01-01T00:00:00Z", "2020-01-01T01:00:00Z")
            pressure = np.array([[100001, 100002, 100003], [100101, 100102, 100103]], dtype="f4")
            arc = root / "arc.nc"
            write_pressure_slice(arc, clock, ids, pressure)
            surface = root / "test_surface.nc"
            with nc.Dataset(surface, "w", format="NETCDF3_CLASSIC") as ds:
                for name, size in (("time", None), ("DateStrLen", 26), ("node", 3), ("nele", 1)):
                    ds.createDimension(name, size)
                _time_variables(ds, clock)
                ds.wind_vector_frame, ds.source_provider, ds.pressure_reference = (
                    "model_xy",
                    "hrrr",
                    "absolute_mean_sea_level",
                )
                ds.source = "FVCOM grid (unstructured) surface forcing"
                ds.source_sha256 = "a" * 64
                ds.model_horizontal_crs = "EPSG:32610"
                ds.createVariable("node_id", "i4", ("node",))[:] = [1, 2, 3]
                ds.createVariable("element_id", "i4", ("nele",))[:] = [1]
                ds.createVariable("lonc", "f8", ("nele",))[:] = [-124]
                ds.createVariable("latc", "f8", ("nele",))[:] = [46]
                for name in ("uwind_stress", "vwind_stress", "air_pressure"):
                    var = ds.createVariable(
                        name, "f4", ("time", "node" if name == "air_pressure" else "nele")
                    )
                    var.units = "Pa"
                    var[:] = pressure if name == "air_pressure" else 1
                c, s = model_basis(CRS.from_epsg(32610), np.array([-124.0]), np.array([46.0]))
                vectors = {
                    "uwind_stress": (c.reshape(1, 1), "Pa"),
                    "vwind_stress": (s.reshape(1, 1), "Pa"),
                }
                _wind_diagnostics_to_earth(ds, vectors)
                np.testing.assert_allclose(vectors["uwind_stress"][0], 1)
                np.testing.assert_allclose(vectors["vwind_stress"][0], 0, atol=1e-10)
            manifest = root / "manifest.json"
            m = dict(
                schema="fvcom_atmosphere_manifest_v1",
                source_index_sha256="a" * 64,
                status="ready",
                provider="hrrr",
                mesh_sha256=sha256_file(grd),
                obc_sha256=sha256_file(obc),
                wind_vector_frame="model_xy",
                pressure_reference="absolute_mean_sea_level",
                start_utc=stamp(clock[0]),
                end_utc=stamp(clock[-1]),
                cadence_seconds=3600,
                records=2,
                files={"surface": record(surface), "obc_pressure": record(arc)},
            )
            write(manifest, m)
            self.assertEqual(verify_bundle(manifest, grd, obc)["records"], 2)
            with nc.Dataset(arc, "r+") as ds:
                ds["air_pressure"][0, 0] = pressure[0, 0]
            m["files"]["obc_pressure"] = record(arc)
            write(manifest, m)
            with self.assertRaisesRegex(ValueError, "OBC slice differs"):
                verify_bundle(manifest, grd, obc)


if __name__ == "__main__":
    unittest.main()
