"""Exercise regional finalization against real small NetCDF fixtures."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd

CONSTITUENTS = ["M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1", "MM", "MF", "M4", "MN4", "MS4", "2N2", "S1", "2Q1", "J1", "L2", "M3", "MU2", "NU2", "OO1"]
START = dt.datetime(2024, 6, 1, tzinfo=dt.timezone.utc)
END = dt.datetime(2024, 6, 4, tzinfo=dt.timezone.utc)
MJD_EPOCH = dt.datetime(1858, 11, 17, tzinfo=dt.timezone.utc)
STARTUP_IINT = 1500
RESTART_FIELDS = ("zeta", "ua", "va", "u", "v", "tauc", "ww", "omega", "temp", "salinity",
                  "viscofm", "viscofh", "km", "kh", "kq", "q2", "q2l", "l")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def chars(variable, values):
    for i, value in enumerate(values):
        variable[i] = np.asarray(list(value.ljust(variable.shape[1])), dtype="S1")


def clock(ds: nc.Dataset, times: list[dt.datetime]) -> None:
    """Independent experiment UTC; the first observed IINT is never the origin."""
    ds.createDimension("DateStrLen", 26)
    chars(ds.createVariable("Times", "S1", ("time", "DateStrLen")),
          [value.strftime("%Y-%m-%dT%H:%M:%S.%f") for value in times])
    ds.createVariable("Itime", "i4", ("time",))[:] = [(value - MJD_EPOCH).days for value in times]
    ds.createVariable("Itime2", "i4", ("time",))[:] = [
        int((value - value.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds() * 1000)
        for value in times]
    floating = ds.createVariable("time", "f4", ("time",))
    floating.units = "days since 1858-11-17 00:00:00"
    floating[:] = nc.date2num(times, floating.units)


def state_file(path: Path, steps, times: list[dt.datetime], *, complete: bool) -> None:
    with nc.Dataset(path, "w") as ds:
        for name, size in (("time", len(times)), ("node", 2), ("nele", 2), ("siglay", 9), ("siglev", 10)):
            ds.createDimension(name, size)
        ds.createVariable("iint", "i4", ("time",))[:] = steps
        clock(ds, times)
        for name in RESTART_FIELDS if complete else ("zeta", "ua", "va"):
            if name == "zeta":
                dimensions = ("time", "node")
            elif name in {"ua", "va", "tauc"}:
                dimensions = ("time", "nele")
            elif name in {"u", "v", "ww"}:
                dimensions = ("time", "siglay", "nele")
            elif name in {"temp", "salinity", "viscofm", "viscofh"}:
                dimensions = ("time", "siglay", "node")
            else:
                dimensions = ("time", "siglev", "node")
            ds.createVariable(name, "f4", dimensions, fill_value=-9999)[:] = (
                20 if name == "temp" else 30 if name == "salinity" else 1)


def fixture(project: Path, mixed: bool, runner, audit_script: Path, skill_dir: Path):
    case = "independent_mesh"
    attempt = project / "run" / case / "production_001"
    inputs = project / "input" / case / "production_001"
    observation = project / "analysis/observations"
    for path in (attempt, inputs, observation):
        path.mkdir(parents=True)
    names = ["profileX9", "9933333"] if mixed else ["9933333", "9422222"]
    roles = ["current", "water_level"] if mixed else ["water_level", "water_level"]
    mapping = {"status": "ready", "stations": [{"station_id": name, "role": role, "cell_id": i + 1} for i, (name, role) in enumerate(zip(names, roles))]}
    mapping_path = inputs / "station_mapping.json"
    write(mapping_path, mapping)
    write(observation / "inventory.json", {"stations": [{"id": name, "role": role, "eligible": True} for name, role in zip(names, roles)]})
    write(observation / "observation_manifest.json", {"status": "ready", "station_inventory": str(observation / "inventory.json"), "hashes": {"station_inventory_sha256": sha256(observation / "inventory.json"), "station_mapping_sha256": sha256(mapping_path)}})
    times = pd.date_range("2024-06-01", "2024-06-04", freq="6min", tz="UTC")
    hours = np.arange(len(times)) / 10
    eta = 0.8 * np.cos(2 * np.pi * hours * 0.0805114007)
    u = 0.5 * np.cos(2 * np.pi * hours * 0.0805114007)
    v = 0.2 * np.sin(2 * np.pi * hours * 0.0805114007)
    for name, role in zip(names, roles):
        if role == "water_level":
            path = observation / "water_level" / f"{name}_noaa_waterlevel.csv"
            path.parent.mkdir(exist_ok=True)
            pd.DataFrame({"time": times[:-1], "observed": eta[:-1] + 0.3, "predicted": eta[:-1] + 0.1}).to_csv(path, index=False)
        else:
            path = observation / "currents" / name / "depth_mean.csv"
            path.parent.mkdir(parents=True)
            pd.DataFrame({"time": times[:-1] + pd.Timedelta(minutes=3), "east_m_s": (u[:-1] + u[1:]) / 2, "north_m_s": (v[:-1] + v[1:]) / 2}).to_csv(path, index=False)
    nml = attempt / "regional_run.nml"
    nml.write_text("START_DATE='2024-06-01 00:00:00',\nEND_DATE='2024-06-04 00:00:00',\n"
                   "EXTSTEP_SECONDS=1.2,\nISPLIT=4,\nSTARTUP_TYPE='hotstart',\n"
                   "STARTUP_FILE='startup.nc',\n"
                   f"INPUT_DIR='{inputs.as_posix()}',\nOBC_ELEVATION_FILE='region_tide.nc',\n"
                   "NC_VELOCITY=.false.,\nNC_VERTICAL_VEL=.false.,\nNC_SALT_TEMP=.false.,\n"
                   "NC_TURBULENCE=.false.,\nOUT_VELOCITY_3D=.false.,\nOUT_SALT_TEMP=.false.,\n",
                   encoding="utf-8")
    # Preserve the original fixture's IINT 1500. An actual, frozen restart now
    # establishes that record at June 1 midnight; 1500*4.8 is not a UTC origin.
    startup = inputs / "startup.nc"
    state_file(startup, [STARTUP_IINT], [START], complete=True)
    nodal_rule = "utide_exact_time_nodal_v1"
    nodal_flags = [False, False, False, False]
    builder_hash = hashlib.sha256(b"synthetic nodal builder provenance fixture").hexdigest()
    with nc.Dataset(inputs / "region_tide.nc", "w") as tide:
        tide.reconstruction_rule_version = nodal_rule
        tide.utide_ngflags_json = json.dumps(nodal_flags)
        tide.builder_sha256 = builder_hash
        tide.utide_version = "synthetic-test"
    exe = hashlib.sha256(b"synthetic frozen executable").hexdigest()
    write(project / "run/build/executable_reuse_binding.json", {"status": "frozen", "executable_sha256": exe})
    write(project / "request.json", {"case_id": "synthetic_regional", "period": {"analysis_start": "2024-06-01T00:00:00Z", "analysis_end": "2024-06-04T00:00:00Z"}, "tpxo": {"expected_constituent_count": 22}})
    write(project / "forcing" / case / "forcing_manifest.json", {"status": "ready", "constituents": CONSTITUENTS,
          "hashes": {"forcing_sha256": sha256(inputs / "region_tide.nc"), "builder_sha256": builder_hash},
          "provenance": {"reconstruction_rule_version": nodal_rule, "utide_ngflags": nodal_flags, "utide_version": "synthetic-test"}})
    files = {path.name: sha256(path) for path in inputs.iterdir()}
    bundle = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    write(attempt / "input_freeze.json", {"schema": "fvcom_input_freeze_v1", "executable_sha256": exe, "input_dir": str(inputs.relative_to(project)), "files": files, "input_bundle_sha256": bundle, "run_namelist_sha256": sha256(nml)})
    write(attempt / "attempt_manifest.json", {"grid_case": case, "input_dir": str(inputs.relative_to(project)), "attempt_namelist_sha256": sha256(nml)})
    station = attempt / "regional_station_timeseries.nc"
    with nc.Dataset(station, "w") as ds:
        ds.createDimension("time", len(times)); ds.createDimension("station", len(names)); ds.createDimension("clen", 20)
        chars(ds.createVariable("name_station", "S1", ("station", "clen")), names)
        ds.createVariable("iint", "i4", ("time",))[:] = np.arange(len(times)) * 75 + STARTUP_IINT
        # FVCOM's real station product writes float32 MJD. Exercise its allowed
        # precision loss while keeping startup-restart UTC as the sole anchor.
        floating = ds.createVariable("time", "f4", ("time",))
        floating.units = "days since 1858-11-17 00:00:00"
        floating[:] = nc.date2num(times.to_pydatetime().tolist(), floating.units)
        for name, field in (("zeta", eta), ("ua", u), ("va", v)):
            ds.createVariable(name, "f4", ("time", "station"))[:] = np.repeat(field[:, None], len(names), axis=1)
    full = attempt / "regional_0001.nc"
    state_file(full, np.arange(25) * 2250 + STARTUP_IINT,
               [START + dt.timedelta(hours=3 * index) for index in range(25)], complete=False)
    restart = attempt / "regional_restart.nc"
    state_file(restart, [55500], [END], complete=True)
    (attempt / "stdout.log").write_text(" ! 55500 2024-06-04T00:00:00.000000\nTADA!\n")
    (attempt / "stderr.log").write_text("")
    command = [sys.executable, str(audit_script), "--stdout", str(attempt / "stdout.log"), "--stderr", str(attempt / "stderr.log"), "--exit-code", "0", "--namelist", str(nml), "--station-netcdf", str(station), "--full-grid-netcdf", str(full), "--restart-netcdf", str(restart), "--startup-restart-netcdf", str(startup), "--station-mapping", str(mapping_path), "--lineage", str(attempt / "input_freeze.json"), "--expected-station-records", "721", "--expected-full-grid-records", "25", "--output", str(attempt / "production_audit.json")]
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    audit = json.loads((attempt / "production_audit.json").read_text())
    assert audit["time_anchor"]["iint_at_start"] == STARTUP_IINT
    assert audit["time_anchor"]["startup_restart_sha256"] == sha256(startup)
    assert audit["full_grid_time"]["exact_clock_verified"]
    assert audit["restart_time"]["iint_last"] == 55500
    assert set(RESTART_FIELDS) <= set(audit["restart_netcdf"]["required_variables"])
    args = runner.parser().parse_args(["--project", str(project), "--grid-case", case, "--attempt", str(attempt)])
    # Source tests must never silently invoke the installed validation kernels.
    args.skill_dir = skill_dir
    return args, inputs, command


def rejects(call, expected):
    try:
        call()
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        assert expected in str(exc), str(exc)
    else:
        raise AssertionError(f"missing rejection: {expected}")


def main():
    options = argparse.ArgumentParser(description=__doc__)
    options.add_argument("--overrideenvsrcdir", type=Path,
                         help="Directory containing reviewed runner/condenser modules")
    options.add_argument("--validation-skill-dir", type=Path,
                         help="Validation skill package supplying prepare/validate kernels for flat review copies")
    cli = options.parse_args()
    source = cli.overrideenvsrcdir.resolve() if cli.overrideenvsrcdir else Path(__file__).resolve().parent
    sys.path.insert(0, str(source))
    runner = importlib.import_module("run_case_validation")
    assert Path(runner.__file__).resolve().parent == source
    # run_case_validation imports the sibling run-control skill. The audit is
    # deliberately not assumed to be beside this validation-skill selftest.
    audit_module = importlib.import_module(runner.parse_namelist.__module__)
    audit_script = Path(audit_module.__file__).resolve()
    if cli.validation_skill_dir:
        skill_dir = cli.validation_skill_dir.resolve()
    else:
        if source.name != "scripts":
            raise ValueError("flat review copies require --validation-skill-dir")
        skill_dir = source.parent
    assert (skill_dir / "scripts/prepare_validation_tables.py").is_file()
    # Keep the reviewed condenser when the kernels live in a separate package.
    sys.path.insert(0, str(source))
    condenser = importlib.import_module("condense_fvcom_station")
    assert Path(condenser.__file__).resolve().parent == source
    run_validation, verify_prerequisites = runner.run_validation, runner.verify_prerequisites
    passed = []
    with tempfile.TemporaryDirectory(prefix="fvcom_finalization_test_") as temporary:
        root = Path(temporary)
        args, inputs, audit_command = fixture(root / "water_only", False, runner, audit_script, skill_dir)
        args.check_only = True
        assert run_validation(args)["workflow_status"] == "not_run"
        passed.append("check_only_with_frozen_startup")
        original = (inputs / "region_tide.nc").read_bytes()
        (inputs / "region_tide.nc").write_bytes(original + b"corrupted")
        rejects(lambda: verify_prerequisites(args), "frozen input changed")
        (inputs / "region_tide.nc").write_bytes(original)
        passed.append("frozen_input_rejection")
        output = args.project / "analysis" / args.grid_case / f"validation_{args.attempt.name}"
        assert not output.exists()
        station = args.attempt / "regional_station_timeseries.nc"
        station_bytes = station.read_bytes()
        station.unlink()
        rejects(lambda: verify_prerequisites(args), "regional_station_timeseries.nc")
        assert not output.exists()
        station.write_bytes(station_bytes)
        passed.append("missing_output_rejection")
        # This offset preserves cadence and length. It must fail even though the
        # old first-IINT normalization could falsely relabel it as midnight.
        with nc.Dataset(station, "a") as ds:
            ds["iint"][:] = ds["iint"][:] + 1
        rejects(lambda: verify_prerequisites(args), "retrieved station files differ")
        failed_audit = subprocess.run(audit_command, capture_output=True, text=True)
        assert failed_audit.returncode != 0
        assert any(fragment in failed_audit.stderr for fragment in
                   ("decoded output coverage", "floating NetCDF time disagrees")), failed_audit.stderr
        assert not output.exists()
        station.write_bytes(station_bytes)
        passed.append("shifted_iint_fails_hash_and_external_clock")
        startup = inputs / "startup.nc"
        startup_bytes = startup.read_bytes()
        with nc.Dataset(startup, "a") as ds:
            ds["iint"][0] = STARTUP_IINT + 1
        rejects(lambda: verify_prerequisites(args), "frozen input changed: startup.nc")
        startup.write_bytes(startup_bytes)
        passed.append("frozen_startup_rejection")
        audit_path = args.attempt / "production_audit.json"
        audit_bytes = audit_path.read_bytes()
        audit = json.loads(audit_bytes)
        audit["time_anchor"]["iint_at_start"] += 1
        write(audit_path, audit)
        rejects(lambda: verify_prerequisites(args), "supplied time anchor disagrees")
        audit_path.write_bytes(audit_bytes)
        passed.append("forged_anchor_rejection")
        audit = json.loads(audit_bytes)
        audit["lineage_manifest_sha256"] = "0" * 64
        write(audit_path, audit)
        rejects(lambda: verify_prerequisites(args), "exact input freeze")
        audit_path.write_bytes(audit_bytes)
        passed.append("audit_lineage_rejection")
        forcing_path = args.project / "forcing" / args.grid_case / "forcing_manifest.json"
        forcing_bytes = forcing_path.read_bytes()
        for mutate, expected in [
            (lambda f: f.pop("provenance"), "predates exact-time nodal"),
            (lambda f: f["provenance"].update(utide_ngflags=[True,False,False,False]), "predates exact-time nodal"),
            (lambda f: f["hashes"].update(builder_sha256="0"*64), "NetCDF and manifest disagree"),
            (lambda f: f["provenance"].update(utide_version="another-version"), "NetCDF and manifest disagree"),
        ]:
            forcing = json.loads(forcing_bytes)
            mutate(forcing);write(forcing_path, forcing)
            rejects(lambda: verify_prerequisites(args), expected)
            assert not output.exists()
            forcing_path.write_bytes(forcing_bytes)
            passed.append("nodal_provenance_rejection_" + str(len(passed)))
        forcing = json.loads(forcing_bytes)
        forcing["constituents"][15] = "T2"
        write(forcing_path, forcing)
        rejects(lambda: run_validation(args), "lacks a forced constituent")
        assert not output.exists()
        forcing_path.write_bytes(forcing_bytes)
        passed.append("unknown_constituent_rejection")
        args.check_only = False
        result = run_validation(args)
        assert result["workflow_status"] == "validation_complete"
        assert result["current_validation_availability"].startswith("unavailable")
        lineage = json.loads((output / "validation_lineage.json").read_text())
        assert lineage["constituents_forced"] == CONSTITUENTS
        assert lineage["input_freeze_sha256"] == sha256(args.attempt / "input_freeze.json")
        assert lineage["run_namelist_sha256"] == sha256(args.attempt / "regional_run.nml")
        assert lineage["model_output_sha256"] == sha256(station)
        condensed = json.loads((output / "condensation_manifest.json").read_text())
        assert condensed["time_reconstruction"]["iint_origin"] == STARTUP_IINT
        assert condensed["time_reconstruction"]["internal_step_seconds"] == 4.8
        product = next(row for row in condensed["products"] if row["role"] == "water_level")
        table = pd.read_csv(product["path"])
        assert pd.Timestamp(table["time"].iloc[0]) == pd.Timestamp(START)
        assert pd.Timestamp(table["time"].iloc[-1]) == pd.Timestamp(END)
        assert len(table) == 721
        assert {row["station_id"] for row in json.loads((output / "validation_summary.json").read_text())["water_level"]} == {"9933333", "9422222"}
        report = (output / "validation_report.html").read_text()
        assert "Current validation is unavailable" in report and "Galveston" not in report
        passed.append("water_only_exact_ids_constituents_and_startup_utc")
        rejects(lambda: run_validation(args), "immutable")
        passed.append("immutable_output_rejection")
        args, _, _ = fixture(root / "mixed", True, runner, audit_script, skill_dir)
        result = run_validation(args)
        assert result["current_validation_availability"] == "available"
        output = args.project / "analysis" / args.grid_case / f"validation_{args.attempt.name}"
        summary = json.loads((output / "validation_summary.json").read_text())
        assert summary["current"][0]["station_id"] == "profileX9"
        assert summary["current"][0]["metrics"]["vector_rmse"] < 1e-6
        passed.append("mixed_current_arbitrary_ids_exact_midpoint_alignment")
    print(json.dumps({"status": "passed", "tests_passed": len(passed), "tests": passed,
                      "runner_module": str(Path(runner.__file__).resolve()), "runner_sha256": sha256(Path(runner.__file__)),
                      "audit_module": str(audit_script), "audit_sha256": sha256(audit_script),
                      "condenser_module": str(Path(condenser.__file__).resolve()),
                      "validation_skill_dir": str(skill_dir)}, indent=2))


if __name__ == "__main__":
    main()
