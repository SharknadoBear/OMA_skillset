"""Exercise the production-audit CLI with clocked cold/hotstart NetCDF fixtures."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import netCDF4 as nc4
import numpy as np

START = dt.datetime(2025, 4, 1, tzinfo=dt.timezone.utc)
END = START + dt.timedelta(minutes=12)
MJD_EPOCH = dt.datetime(1858, 11, 17, tzinfo=dt.timezone.utc)
OUTPUT_FLAGS = ("NC_VELOCITY", "NC_VERTICAL_VEL", "NC_SALT_TEMP", "NC_TURBULENCE",
                "OUT_VELOCITY_3D", "OUT_SALT_TEMP")
# The frozen FVCOM restart constructor always writes these fields, irrespective
# of the optional station/full-grid output flags. Keep this fixture independent
# of required_fields(), so an accidentally weakened implementation cannot pass.
RESTART_FIELDS = ("zeta", "ua", "va", "u", "v", "tauc", "ww", "omega", "temp", "salinity",
                  "viscofm", "viscofh", "km", "kh", "kq", "q2", "q2l", "l",
                  "et", "rho1", "rmean1", "tmean1", "smean1", "cor", "cc_sponge")
# FVCOM-GitHub/FVCOM release-v4.3.1, commit e2a265ca0c3f8a514a026906afec35c428c8d8a7:
# src/mod_ncdio.F::WET_DRY_FILE_OBJECT writes only these five wet/dry fields;
# src/mod_startup.F reads the same five. No native inundation_cells field exists.
CONDITIONAL_RESTART_FIELDS = ("obc_nodes", "obc_type", "wet_nodes", "wet_cells", "wet_nodes_prev_int",
                              "wet_cells_prev_int", "wet_cells_prev_ext")
INTEGER_RESTART_FIELDS = set(CONDITIONAL_RESTART_FIELDS)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_chars(variable: nc4.Variable, rows: list[str]) -> None:
    for index, value in enumerate(rows):
        variable[index, :] = np.asarray(list(value.ljust(variable.shape[1])), dtype="S1")


def clock(ds: nc4.Dataset, times: list[dt.datetime]) -> None:
    """UTC comes from the experiment, never from the observed first IINT."""
    ds.createDimension("DateStrLen", 26)
    write_chars(ds.createVariable("Times", "S1", ("time", "DateStrLen")),
                [value.strftime("%Y-%m-%dT%H:%M:%S.%f") for value in times])
    ds.createVariable("Itime", "i4", ("time",))[:] = [(value - MJD_EPOCH).days for value in times]
    ds.createVariable("Itime2", "i4", ("time",))[:] = [
        int((value - value.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds() * 1000)
        for value in times]
    floating = ds.createVariable("time", "f4", ("time",))
    floating.units = "days since 1858-11-17 00:00:00"
    floating[:] = nc4.date2num(times, floating.units)


def field(ds: nc4.Dataset, name: str) -> None:
    if name in {"obc_nodes", "obc_type"}:
        dimensions = ("nobc",)
    elif name in {"cor", "cc_sponge"}:
        dimensions = ("nele",)
    elif name in {"zeta", "et", "wet_nodes", "wet_nodes_prev_int"}:
        dimensions = ("time", "node")
    elif name in {"rmean1", "tmean1", "smean1"}:
        dimensions = ("siglay", "node")
    elif name in {"ua", "va", "tauc", "wet_cells", "wet_cells_prev_int", "wet_cells_prev_ext"}:
        dimensions = ("time", "nele")
    elif name in {"u", "v", "ww", "viscofm"}:
        dimensions = ("time", "siglay", "nele")
    elif name in {"temp", "salinity", "viscofh", "rho1"}:
        dimensions = ("time", "siglay", "node")
    else:
        dimensions = ("time", "siglev", "node")
    ds.createVariable(name, "i4" if name in INTEGER_RESTART_FIELDS else "f4", dimensions, fill_value=-9999)[:] = (
        20 if name == "temp" else 30 if name == "salinity" else 1)


def state_file(path: Path, steps: list[int], times: list[dt.datetime], *, complete: bool) -> None:
    with nc4.Dataset(path, "w") as ds:
        for name, size in (("time", len(steps)), ("node", 2), ("nele", 1), ("siglay", 9), ("siglev", 10), ("nobc", 1)):
            ds.createDimension(name, size)
        ds.createVariable("iint", "i4", ("time",))[:] = steps
        clock(ds, times)
        for name in RESTART_FIELDS + CONDITIONAL_RESTART_FIELDS if complete else ("zeta", "ua", "va"):
            field(ds, name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overrideenvsrcdir", type=Path,
                        help="Review directory containing the audit and time-anchor modules")
    args = parser.parse_args()
    source = args.overrideenvsrcdir.resolve() if args.overrideenvsrcdir else Path(__file__).resolve().parent
    sys.path.insert(0, str(source))
    audit_module = importlib.import_module("audit_fvcom_production")
    script = Path(audit_module.__file__).resolve()
    assert script.parent == source, f"unexpected audit import: {script}"
    passed: list[str] = []

    with tempfile.TemporaryDirectory(prefix="fvcom_production_audit_test_") as temporary:
        root = Path(temporary)
        (root / "stdout.log").write_text("2025-04-01T00:12:00.000000\nTADA!\n", encoding="utf-8")
        (root / "stderr.log").write_text("", encoding="utf-8")
        nml, mapping, lineage = root / "run.nml", root / "mapping.json", root / "input_freeze.json"
        mapping.write_text(json.dumps({"stations": [{"station_id": "station_a"}]}), encoding="utf-8")
        startup, station, full, restart = (root / name for name in ("startup.nc", "station.nc", "full.nc", "restart.nc"))
        # A nonzero origin is supplied by the actual startup file, not inferred
        # from station data or divided by the current run's timestep.
        state_file(startup, [500000], [START], complete=True)

        def write_nml(kind: str, *, enabled: tuple[str, ...] = (), omit: str | None = None) -> None:
            values = ["START_DATE = '2025-04-01 00:00:00',", "END_DATE = '2025-04-01 00:12:00',",
                      "EXTSTEP_SECONDS = 1.2,", "ISPLIT = 2,", f"STARTUP_TYPE = '{kind}',",
                      "STARTUP_FILE = 'startup.nc',", "INPUT_DIR = '.',", "OBC_ON = T,", "WETTING_DRYING_ON = T,"]
            values += [f"{flag} = {'.true.' if flag in enabled else '.false.'},"
                       for flag in OUTPUT_FLAGS if flag != omit]
            nml.write_text("\n".join(values) + "\n", encoding="utf-8")
            freeze()

        def freeze() -> None:
            files = {"startup.nc": sha256(startup), "mapping.json": sha256(mapping)}
            lineage.write_text(json.dumps({"schema": "fvcom_input_freeze_v1", "files": files,
                "input_bundle_sha256": hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                "run_namelist_sha256": sha256(nml)}), encoding="utf-8")

        def outputs(origin: int) -> None:
            with nc4.Dataset(station, "w") as ds:
                for name, size in (("time", 3), ("station", 1), ("clen", 20), ("siglay", 9)):
                    ds.createDimension(name, size)
                write_chars(ds.createVariable("name_station", "S1", ("station", "clen")), ["station_a"])
                ds.createVariable("iint", "i4", ("time",))[:] = [origin, origin + 150, origin + 300]
                floating = ds.createVariable("time", "f4", ("time",))
                floating.units = "days since 1858-11-17 00:00:00"
                floating[:] = nc4.date2num([START, START + dt.timedelta(minutes=6), END], floating.units)
                for name in ("zeta", "ua", "va"):
                    ds.createVariable(name, "f4", ("time", "station"), fill_value=-9999)[:] = 1
            state_file(full, [origin, origin + 150, origin + 300], [START, START + dt.timedelta(minutes=6), END], complete=False)
            state_file(restart, [origin + 300], [END], complete=True)

        output = root / "audit.json"
        command = [sys.executable, str(script), "--stdout", str(root / "stdout.log"), "--stderr", str(root / "stderr.log"),
                   "--exit-code", "0", "--namelist", str(nml), "--station-netcdf", str(station),
                   "--full-grid-netcdf", str(full), "--restart-netcdf", str(restart),
                   "--startup-restart-netcdf", str(startup), "--station-mapping", str(mapping), "--lineage", str(lineage),
                   "--expected-station-records", "3", "--expected-full-grid-records", "3",
                   "--station-interval-seconds", "360", "--full-grid-interval-seconds", "360", "--output", str(output)]

        def succeeds(label: str, cmd: list[str] | None = None) -> dict:
            completed = subprocess.run(cmd or command, capture_output=True, text=True)
            assert completed.returncode == 0, completed.stderr
            result = json.loads(output.read_text(encoding="utf-8"))
            assert result["status"] == "passed"
            assert result["audit_rules_version"] == "startup_anchor_exact_clock_3d_v1"
            assert result["station_time"]["decoded_end_utc"] == "2025-04-01T00:12:00Z"
            assert result["numerics"]["internal_step_seconds"] == 2.4
            assert result["station_time"]["interval_seconds"] == 360
            assert result["station_time"]["native_station_clock_verified"]
            assert result["station_netcdf"][0]["required_variables_finite_and_unmasked"]
            assert result["full_grid_time"]["exact_clock_verified"]
            assert result["restart_time"]["exact_clock_verified"]
            assert set(RESTART_FIELDS) <= set(result["restart_netcdf"]["required_variables"])
            assert result["restart_netcdf"]["required_native_dimensions_verified"]
            passed.append(label)
            return result

        def fails(label: str, fragment: str, cmd: list[str] | None = None) -> None:
            # A rejected invocation must not replace a prior passing artifact.
            prior = output.read_bytes() if output.exists() else None
            completed = subprocess.run(cmd or command, capture_output=True, text=True)
            assert completed.returncode != 0 and fragment in completed.stderr, completed.stderr
            if prior is not None:
                assert output.read_bytes() == prior
            passed.append(label)

        write_nml("coldstart")
        outputs(0)
        result = succeeds("coldstart_exact_clocks_complete_restart")
        assert result["time_anchor"]["method"] == "coldstart_zero" and result["time_anchor"]["iint_at_start"] == 0
        with nc4.Dataset(restart) as ds:
            assert "inundation_cells" not in ds.variables
        assert "inundation_cells" not in result["restart_netcdf"]["required_variables"]
        assert set(CONDITIONAL_RESTART_FIELDS) <= set(result["restart_netcdf"]["required_variables"])
        passed.append("native_431_wet_restart_without_inundation_cells")

        # An extension field is optional, but known physical data must still be
        # finite and unmasked when present. Its values need not be binary.
        native_restart_bytes = restart.read_bytes()
        with nc4.Dataset(restart, "a") as ds:
            ds.createVariable("inundation_cells", "i4", ("time", "nele"), fill_value=-9999)[:] = 2
        result = succeeds("optional_inundation_cells_nonbinary")
        assert "inundation_cells" in result["restart_netcdf"]["checked_variables"]
        assert "inundation_cells" not in result["restart_netcdf"]["required_variables"]
        with nc4.Dataset(restart, "a") as ds:
            ds["inundation_cells"][:] = -9999
        fails("optional_inundation_cells_masked", "non-finite variables ['inundation_cells']")
        restart.write_bytes(native_restart_bytes)
        with nc4.Dataset(station, "a") as ds:
            ds.renameVariable("time", "removed_time")
        fails("station_native_time_missing", "no native clock")
        with nc4.Dataset(station, "a") as ds:
            clock(ds, [START, START + dt.timedelta(minutes=6), END])
            ds.renameVariable("time", "removed_new_time")
        fails("station_exact_only_missing_native_time", "missing time")
        outputs(0)
        with nc4.Dataset(station, "a") as ds:
            ds["iint"][:] = [1, 151, 301]
        fails("coldstart_shifted_iint", "decoded output coverage")

        write_nml("hotstart")
        outputs(500000)
        result = succeeds("hotstart_frozen_independent_origin")
        assert result["time_anchor"]["iint_at_start"] == 500000
        assert result["time_anchor"]["startup_restart_sha256"] == sha256(startup)
        assert result["station_time"]["iint_first"] == 500000
        # Preserve the finite-fill, irregular cadence, and overlapping-stack tests.
        with nc4.Dataset(station, "a") as ds:
            ds["zeta"][1, 0] = -9999
        fails("masked_finite_fill", "non-finite")
        with nc4.Dataset(station, "a") as ds:
            ds["zeta"][1, 0] = 1
            ds["iint"][:] = [500000, 500151, 500300]
        fails("irregular_cadence", "cadence")
        with nc4.Dataset(station, "a") as ds:
            ds["iint"][:] = [500001, 500151, 500301]
        fails("hotstart_shifted_iint", "decoded output coverage")
        with nc4.Dataset(station, "a") as ds:
            ds["iint"][:] = [500000, 500150, 500300]
        duplicate = root / "duplicate.nc"
        shutil.copyfile(station, duplicate)
        repeated = command + ["--station-netcdf", str(duplicate)]
        assert succeeds("equal_overlap", repeated)["station_time"]["verified_duplicate_boundary_records"] == 3
        with nc4.Dataset(duplicate, "a") as ds:
            ds["ua"][1, 0] = 2
        fails("conflicting_overlap", "conflicting ua", repeated)

        original_startup = startup.read_bytes()
        with nc4.Dataset(startup, "a") as ds:
            ds["iint"][0] = 499999
        fails("startup_frozen_hash", "startup restart differs from frozen")
        startup.write_bytes(original_startup)
        with nc4.Dataset(restart, "a") as ds:
            ds["iint"][0] = 500301
        fails("restart_iint_against_exact_clock", "exact NetCDF clock disagrees")
        with nc4.Dataset(restart, "a") as ds:
            ds["iint"][0] = 500300
            ds["omega"][0, 0, 0] = -9999
        fails("masked_restart_3d", "non-finite")
        with nc4.Dataset(restart, "a") as ds:
            ds["omega"][0, 0, 0] = 1
            ds.renameVariable("q2l", "missing_q2l")
        fails("missing_restart_turbulence", "missing variables ['q2l']")
        with nc4.Dataset(restart, "a") as ds:
            ds.renameVariable("missing_q2l", "q2l")

        # Source-native restart extras remain required even when all optional
        # station/full-grid output flags are false. Static mean fields deliberately
        # have no time dimension, matching the frozen restart writer.
        for name in ("et", "rho1", "rmean1", "tmean1", "smean1", "cor", "cc_sponge") + CONDITIONAL_RESTART_FIELDS:
            with nc4.Dataset(restart, "a") as ds:
                ds.renameVariable(name, "missing_" + name)
            fails("missing_restart_" + name, f"missing variables ['{name}']")
            with nc4.Dataset(restart, "a") as ds:
                ds.renameVariable("missing_" + name, name)
                ds[name][...] = -9999
            fails("masked_restart_" + name, f"non-finite variables ['{name}']")
            if name not in INTEGER_RESTART_FIELDS:
                with nc4.Dataset(restart, "a") as ds:
                    ds[name][...] = np.nan
                fails("nonfinite_restart_" + name, f"non-finite variables ['{name}']")
            with nc4.Dataset(restart, "a") as ds:
                ds[name][...] = 1
        succeeds("restored_unconditional_restart_state")

        complete_restart_bytes = restart.read_bytes()
        for name in ("et", "rho1", "rmean1", "tmean1", "smean1", "cor", "cc_sponge") + CONDITIONAL_RESTART_FIELDS:
            with nc4.Dataset(restart, "a") as ds:
                ds.renameVariable(name, "wrong_shape_original_" + name)
                ds.createVariable(name, "f4", ("time",))[:] = 1
            fails("wrong_native_restart_dimensions_" + name, "native dimensions for " + name)
            restart.write_bytes(complete_restart_bytes)

        # OBC and wet/dry requirements follow explicit runtime flags; ordinary
        # station/full-grid output flags do not control restart completeness.
        restart_bytes = restart.read_bytes()
        with nc4.Dataset(restart, "a") as ds:
            for name in CONDITIONAL_RESTART_FIELDS:
                ds.renameVariable(name, "disabled_" + name)
        enabled_nml = nml.read_bytes()
        nml.write_text(nml.read_text().replace("OBC_ON = T", "OBC_ON = F").replace("WETTING_DRYING_ON = T", "WETTING_DRYING_ON = F"))
        freeze()
        succeeds("disabled_obc_wet_state_not_required")
        for flag in ("OBC_ON", "WETTING_DRYING_ON"):
            nml.write_text("\n".join(line for line in enabled_nml.decode().splitlines() if not line.startswith(flag + " =")))
            freeze()
            fails("explicit_restart_" + flag, "explicit logical " + flag)
        nml.write_bytes(enabled_nml)
        restart.write_bytes(restart_bytes)
        freeze()

        full_bytes = full.read_bytes()
        with nc4.Dataset(full, "a") as ds:
            ds["Itime2"][1] = 360001
        fails("full_integer_clock_disagrees", "Times disagrees with Itime/Itime2")
        full.write_bytes(full_bytes)
        with nc4.Dataset(full, "a") as ds:
            ds.renameVariable("Times", "missing_Times")
        fails("full_exact_clock_required", "exact FVCOM clock is missing")
        full.write_bytes(full_bytes)

        write_nml("hotstart", enabled=("NC_VELOCITY", "NC_VERTICAL_VEL", "NC_SALT_TEMP", "NC_TURBULENCE"))
        fails("enabled_full_fields_missing", "missing variables")
        state_file(full, [500000, 500150, 500300], [START, START + dt.timedelta(minutes=6), END], complete=True)
        succeeds("enabled_full_3d_fields")
        with nc4.Dataset(full, "a") as ds:
            ds["u"][1, 0, 0] = np.nan
        fails("enabled_full_3d_nonfinite", "non-finite")
        with nc4.Dataset(full, "a") as ds:
            ds["u"][1, 0, 0] = 1
        write_nml("hotstart", enabled=OUTPUT_FLAGS)
        fails("enabled_station_fields_missing", "missing variables")
        with nc4.Dataset(station, "a") as ds:
            for name in ("u", "v", "ww", "temp", "salinity"):
                ds.createVariable(name, "f4", ("time", "siglay", "station"), fill_value=-9999)[:] = 1
        succeeds("enabled_station_3d_fields")
        shutil.copyfile(station, duplicate)
        with nc4.Dataset(duplicate, "a") as ds:
            ds["u"][1, 0, 0] = 2
        fails("conflicting_3d_overlap", "conflicting u", repeated)
        write_nml("hotstart", omit="NC_TURBULENCE")
        fails("explicit_output_flags_required", "explicit logical NC_TURBULENCE")
        write_nml("hotstart")
        nml_bytes = nml.read_bytes()
        nml.write_bytes(nml_bytes + b"! mutated after freeze\n")
        fails("namelist_lineage_mismatch", "run-namelist hash differs")
        nml.write_bytes(nml_bytes)
        mapping_bytes = mapping.read_bytes()
        mapping.write_bytes(mapping_bytes + b"\n")
        fails("mapping_lineage_mismatch", "station-mapping hash differs")
        mapping.write_bytes(mapping_bytes)
        succeeds("all_restored_inputs_pass")

    print(json.dumps({"status": "passed", "tests_passed": len(passed), "audit_module": str(script),
                      "audit_module_sha256": sha256(script), "tests": passed}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
