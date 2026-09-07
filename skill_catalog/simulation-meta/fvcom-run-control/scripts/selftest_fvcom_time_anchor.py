"""Portable synthetic regression tests for sibling fvcom_time_anchor.py.

Run with the skill's scientific Python environment. All NetCDF fixtures are
temporary; no campaign paths, source archives, or model outputs are required.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
import tempfile
import unittest

import netCDF4 as nc4
import numpy as np

from fvcom_time_anchor import (
    MJD_EPOCH, anchored_times, audit_restart, check_file_clock, exact_clock,
    iso, parse_utc, read_iint, resolve_anchor, restart_anchor, sha256,
)


class TimeAnchorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="fvcom_time_anchor_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.start = parse_utc("2025-04-01T00:00:00Z")
        self.times = [self.start + dt.timedelta(seconds=360 * index) for index in range(3)]
        self.steps = np.asarray([75000, 75150, 75300], dtype=np.int64)
        self.step_seconds = 2.4
        self.startup = self.write_nc("startup.nc", [75000], [self.start])
        self.anchor = restart_anchor(self.startup, self.start, sha256(self.startup))
        self.namelist_path = self.root / "run.nml"
        self.values = {
            "START_DATE": iso(self.start), "STARTUP_TYPE": "hotstart",
            "STARTUP_FILE": self.startup.name, "INPUT_DIR": ".",
        }

    def write_nc(self, name: str, steps, times, *, exact: bool = True,
                 float_time: bool = True, omit: tuple[str, ...] = (),
                 iint_dtype: str = "i4") -> Path:
        path = self.root / name
        with nc4.Dataset(path, "w") as ds:
            ds.createDimension("time", len(steps))
            ds.createDimension("DateStrLen", 26)
            if "iint" not in omit:
                ds.createVariable("iint", iint_dtype, ("time",), fill_value=-9999)[:] = steps
            if exact:
                if "Times" not in omit:
                    var = ds.createVariable("Times", "S1", ("time", "DateStrLen"))
                    for index, time in enumerate(times):
                        var[index] = np.asarray(list(time.strftime("%Y-%m-%dT%H:%M:%S.%f")), dtype="S1")
                if "Itime" not in omit:
                    ds.createVariable("Itime", "i4", ("time",))[:] = [(time - MJD_EPOCH).days for time in times]
                if "Itime2" not in omit:
                    ms = [int((time - time.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds() * 1000) for time in times]
                    ds.createVariable("Itime2", "i4", ("time",))[:] = ms
            if float_time:
                var = ds.createVariable("time", "f4", ("time",), fill_value=-9999)
                var.units = "days since 1858-11-17 00:00:00"
                var[:] = nc4.date2num(times, var.units)
        return path

    def check_path(self, path: Path, *, require_exact: bool = True):
        with nc4.Dataset(path) as ds:
            expected = anchored_times(read_iint(ds), self.anchor, self.step_seconds)
            return check_file_clock(ds, expected, require_exact=require_exact)

    def test_coldstart_zero_origin(self):
        values = {"START_DATE": iso(self.start), "STARTUP_TYPE": "coldstart"}
        anchor = resolve_anchor(values, self.namelist_path)
        self.assertEqual(anchor["iint_at_start"], 0)
        self.assertEqual(anchor["method"], "coldstart_zero")
        self.assertEqual(anchored_times(np.asarray([0, 150, 300]), anchor, 2.4), self.times)

    def test_hotstart_resolves_relative_input_and_hash(self):
        anchor = resolve_anchor(self.values, self.namelist_path, expected_sha256=sha256(self.startup))
        self.assertEqual(anchor["iint_at_start"], 75000)
        self.assertEqual(anchor["startup_restart_sha256"], sha256(self.startup))
        self.assertEqual(anchor["startup_restart_record_index"], 0)
        self.assertEqual(anchored_times(self.steps, anchor, self.step_seconds), self.times)

    def test_hotstart_explicit_relocated_input(self):
        values = dict(self.values, INPUT_DIR="absent_directory", STARTUP_FILE="relocated.nc")
        anchor = resolve_anchor(values, self.namelist_path, startup_restart=self.startup,
                                expected_sha256=sha256(self.startup))
        self.assertEqual(anchor["iint_at_start"], 75000)

    def test_select_startup_record_by_exact_time(self):
        path = self.write_nc("stack.nc", [74900, 75000], [self.start - dt.timedelta(seconds=240), self.start])
        anchor = restart_anchor(path, self.start)
        self.assertEqual(anchor["startup_restart_record_index"], 1)
        self.assertEqual(anchor["iint_at_start"], 75000)

    def test_wrong_startup_time_rejected(self):
        with self.assertRaisesRegex(ValueError, "START_DATE"):
            restart_anchor(self.startup, self.start + dt.timedelta(seconds=1))

    def test_wrong_startup_hash_rejected(self):
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            restart_anchor(self.startup, self.start, "0" * 64)

    def test_supplied_anchor_rechecked(self):
        actual = resolve_anchor(self.values, self.namelist_path, supplied=self.anchor)
        self.assertEqual(actual["iint_at_start"], 75000)
        with self.assertRaisesRegex(ValueError, "supplied time anchor"):
            resolve_anchor(self.values, self.namelist_path, supplied=dict(self.anchor, iint_at_start=75001))

    def test_missing_startup_semantics_rejected(self):
        with self.assertRaisesRegex(ValueError, "explicit hotstart or coldstart"):
            resolve_anchor({"START_DATE": iso(self.start)}, self.namelist_path)

    def test_missing_hotstart_file_rejected(self):
        with self.assertRaisesRegex(ValueError, "STARTUP_FILE"):
            resolve_anchor({"START_DATE": iso(self.start), "STARTUP_TYPE": "hotstart"}, self.namelist_path)

    def test_exact_output_clock_passes(self):
        path = self.write_nc("full.nc", self.steps, self.times)
        self.assertTrue(self.check_path(path)["exact_clock_verified"])

    def test_shifted_iint_is_not_rebased(self):
        path = self.write_nc("shift.nc", self.steps + 1, self.times)
        shifted = anchored_times(self.steps + 1, self.anchor, self.step_seconds)
        self.assertEqual(shifted[0], self.start + dt.timedelta(seconds=2.4))
        with self.assertRaisesRegex(ValueError, "startup-anchored iint"):
            self.check_path(path)

    def test_coherently_shifted_absolute_clocks_rejected(self):
        times = [value + dt.timedelta(seconds=1) for value in self.times]
        path = self.write_nc("shift_clocks.nc", self.steps, times)
        with self.assertRaisesRegex(ValueError, "startup-anchored iint"):
            self.check_path(path)

    def test_contradictory_Times_integer_clock_rejected(self):
        path = self.write_nc("contradiction.nc", self.steps, self.times)
        with nc4.Dataset(path, "a") as ds:
            ds["Itime2"][1] += 1
        with self.assertRaisesRegex(ValueError, "Times disagrees with Itime/Itime2"):
            self.check_path(path)

    def test_invalid_milliseconds_of_day_rejected(self):
        path = self.write_nc("bad_ms.nc", self.steps, self.times)
        with nc4.Dataset(path, "a") as ds:
            ds["Itime2"][0] = 86400000
        with self.assertRaisesRegex(ValueError, "milliseconds of day"):
            self.check_path(path)

    def test_missing_exact_clock_rejected_when_required(self):
        path = self.write_nc("station.nc", self.steps, self.times, exact=False)
        with self.assertRaisesRegex(ValueError, "exact FVCOM clock is missing"):
            self.check_path(path)

    def test_partial_exact_clock_rejected_even_if_optional(self):
        path = self.write_nc("partial.nc", self.steps, self.times, omit=("Itime2",))
        with self.assertRaisesRegex(ValueError, "exact FVCOM clock is missing"):
            self.check_path(path, require_exact=False)

    def test_float32_MJD_quantization_is_accepted(self):
        path = self.write_nc("float_station.nc", self.steps, self.times, exact=False)
        result = self.check_path(path, require_exact=False)
        self.assertFalse(result["exact_clock_verified"])
        self.assertGreater(result["maximum_float_time_error_seconds"], 1.0)
        # Half an ULP near this modern MJD is 168.75 seconds.
        self.assertLessEqual(result["maximum_float_time_error_seconds"], 168.75)

    def test_float_time_outside_precision_rejected(self):
        path = self.write_nc("wrong_float.nc", self.steps, self.times, exact=False)
        with nc4.Dataset(path, "a") as ds:
            raw = np.float32(ds["time"][1])
            ds["time"][1] = raw + 2 * np.spacing(raw)
        with self.assertRaisesRegex(ValueError, "beyond its precision"):
            self.check_path(path, require_exact=False)

    def test_masked_iint_rejected(self):
        path = self.write_nc("masked.nc", [75000, -9999, 75300], self.times)
        with self.assertRaisesRegex(ValueError, "masked or nonfinite"):
            self.check_path(path)

    def test_nonfinite_iint_rejected(self):
        path = self.write_nc("nan.nc", [75000, np.nan, 75300], self.times, iint_dtype="f8")
        with self.assertRaisesRegex(ValueError, "masked or nonfinite"):
            self.check_path(path)

    def test_fractional_iint_rejected(self):
        path = self.write_nc("fraction.nc", [75000, 75150.5, 75300], self.times, iint_dtype="f8")
        with self.assertRaisesRegex(ValueError, "integer steps"):
            self.check_path(path)

    def test_nonmonotonic_iint_rejected(self):
        path = self.write_nc("backwards.nc", [75000, 74999, 75300], self.times)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            self.check_path(path)

    def test_missing_iint_rejected(self):
        path = self.write_nc("missing_iint.nc", self.steps, self.times, omit=("iint",))
        with self.assertRaisesRegex(ValueError, "missing iint"):
            self.check_path(path)

    def test_restart_exact_end_passes(self):
        path = self.write_nc("restart.nc", [75300], [self.times[-1]])
        result = audit_restart(path, self.anchor, self.step_seconds, self.times[-1])
        self.assertTrue(result["exact_clock_verified"])
        self.assertEqual(result["iint_last"], 75300)

    def test_restart_wrong_iint_rejected(self):
        path = self.write_nc("wrong_restart.nc", [75301], [self.times[-1]])
        with self.assertRaisesRegex(ValueError, "startup-anchored iint"):
            audit_restart(path, self.anchor, self.step_seconds, self.times[-1])

    def test_restart_wrong_requested_end_rejected(self):
        path = self.write_nc("short_restart.nc", [75300], [self.times[-1]])
        with self.assertRaisesRegex(ValueError, "exact requested END_DATE"):
            audit_restart(path, self.anchor, self.step_seconds, self.times[-1] + dt.timedelta(seconds=1))

    def test_fractional_second_restart_mismatch_rejected(self):
        path = self.write_nc("fractional_restart.nc", [75300], [self.times[-1] + dt.timedelta(milliseconds=100)])
        with self.assertRaisesRegex(ValueError, "startup-anchored iint"):
            audit_restart(path, self.anchor, self.step_seconds, self.times[-1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
