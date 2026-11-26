"""Local analytic checks for the lake experiment diagnostics. Generates no persistent model output."""

import csv
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import netCDF4 as nc
import numpy as np

from lake_physics import member, setup, sha, spinup, thermal_measurement, thermal_settings, utc
from lake_budget import budget

START = datetime(2004, 8, 1, tzinfo=timezone.utc)
DAYS = (3, 6, 9)


class NativeDiagnostics(unittest.TestCase):
    def setUp(self):
        t = tempfile.TemporaryDirectory()
        self.addCleanup(t.cleanup)
        self.d = Path(t.name)
        self.h = 12.0
        self.area = 100.0
        self.times = [START + timedelta(hours=6 * i) for i in range(13)]
        self.force = {}
        for name in ("pe", "atmosphere", "river"):
            path = self.d / (name + ".nc")
            with nc.Dataset(path, "w") as f:
                forcing_times = [START + timedelta(hours=i) for i in range(-24 * max(DAYS),73)]
                self.clock(f, forcing_times)
                f.createDimension("node", 3)
                if name == "river":
                    f.createDimension("rivers", 2)
                    f.createDimension("namelen", 2)
                    text = f.createVariable("river_names", "S1", ("rivers", "namelen"))
                    text[:] = np.asarray([["r", "1"], ["r", "2"]], dtype="S1")
                    v = f.createVariable("river_flux", "f8", ("time", "rivers"))
                    v[:] = [[1, -1]] * len(forcing_times)
                    v = f.createVariable("river_temp", "f8", ("time", "rivers"))
                    v[:] = [[10, 0]] * len(forcing_times)
                else:
                    for key in ("precip", "evap") if name == "pe" else ("net_heat_flux", "short_wave"):
                        f.createVariable(key, "f8", ("time", "node"))[:] = 0.0
                    if name == "atmosphere":
                        air = f.createVariable("air_pressure", "f8", ("time","node"))
                        air.units = "Pa"
                        air[:] = 101325.
            self.force[name] = self.ref(path)
        self.surface_csv = self.d / "surface_budget.csv"
        with self.surface_csv.open("w", newline="") as stream:
            w = csv.writer(stream)
            w.writerow(["start_utc", "end_utc", "rain_m3", "evaporated_m3", "condensed_m3", "net_heat_j"])
            for k in range(72):
                a = START + timedelta(hours=k)
                w.writerow([a.isoformat(), (a + timedelta(hours=1)).isoformat(), 0, 0, 0, 0])
        self.surf = self.js(
            "surface.json",
            dict(
                files={
                    r["path"]: r["sha256"]
                    for r in [self.force["pe"], self.force["atmosphere"], self.ref(self.surface_csv)]
                }
            ),
        )
        self.riv = self.js(
            "rivers.json",
            dict(
                count=len(forcing_times),
                files={"river.nc": self.force["river"]["sha256"]},
                entries=[
                    dict(index=1, cell_id=1, edge_nodes=[1, 2], river_name="r1"),
                    dict(index=2, cell_id=1, edge_nodes=[2, 3], river_name="r2"),
                ],
            ),
        )
        self.freeze = self.js("freeze.json", dict(executable_sha256="a" * 64))
        self.members = {str(day): self.gen(day, 0.4 if day == DAYS[0] else 0.0) for day in DAYS}
        self.geometry = self.members[str(DAYS[-1])]["outputs"][0]
        self.geom = setup(self.d, {"geometry": self.geometry})
        self.plan = self.js(
            "plan.json",
            dict(
                comparison_start_utc=START.isoformat(),
                comparison_end_utc=self.times[-1].isoformat(),
                stages=[
                    dict(
                        name=f"spinup_{d}d",
                        start_utc=(START - timedelta(days=d)).isoformat(),
                        end_utc=self.times[-1].isoformat(),
                    )
                    for d in DAYS
                ],
                convergence=dict(
                    volume_weighted_temperature_rms_c=0.25,
                    upper_20m_temperature_rms_c=0.5,
                    median_thermocline_abs_difference_m=2.0,
                    lake_mean_elevation_difference_m=0.02,
                ),
            ),
        )

    def clock(self, f, dates):
        f.createDimension("time", len(dates))
        epoch = START.replace(year=2000, month=1, day=1)
        f.createVariable("Itime", "i4", ("time",))[:] = [(d - epoch).days for d in dates]
        f["Itime"].units = "days since 2000-01-01 00:00:00"
        f.createVariable("Itime2", "i4", ("time",))[:] = [(d - epoch).seconds * 1000 for d in dates]

    def ref(self, p):
        return dict(path=p.name, sha256=sha(p))

    def js(self, name, data):
        p = self.d / name
        p.write_text(json.dumps(data))
        return self.ref(p)

    def gen(self, day, offset, *, full_window=False, cold=None):
        path = self.d / f"output{day}.nc"
        dates = ([START-timedelta(hours=6*i) for i in range(day*4,0,-1)] if full_window else []) + self.times
        with nc.Dataset(path, "w") as f:
            f.CoordinateSystem = "GeoReferenced"
            self.clock(f, dates)
            for k, size in (("node", 3), ("nele", 1), ("siglay", 4), ("siglev", 5), ("three", 3)):
                f.createDimension(k, size)
            for k, x in (
                ("h", [self.h] * 3),
                ("lon", [-90, -90, -89.998]),
                ("lat", [47, 47.002, 47]),
                ("art1", [self.area] * 3),
            ):
                f.createVariable(k, "f8", ("node",))[:] = x
            f.createVariable("nv", "i4", ("three", "nele"))[:] = [[1], [2], [3]]
            f.createVariable("siglev", "f8", ("siglev", "node"))[:] = np.repeat(
                -np.arange(5)[:, None] / 4, 3, axis=1
            )
            for k in (
                "temp",
                "salinity",
                "u",
                "v",
                "wet_nodes",
                "wet_cells",
                "zeta",
                "net_heat_flux",
                "short_wave",
            ):
                a = "nele" if k in ("u", "v", "wet_cells") else "node"
                vertical = ("siglay",) if k in ("u", "v", "temp", "salinity") else ()
                f.createVariable(k, "f8", ("time", *vertical, a))[:] = int("wet" in k)
            f["temp"][:] = np.broadcast_to(
                np.array([16, 12, 8, 4])[None, :, None] + offset, (len(dates), 4, 3)
            )
            if cold is not None:
                f["temp"][1:3,0,0] = cold
        nml = self.d / f"stage{day}.nml"
        nml.write_text(
            "\n".join(
                [
                    "GRID_FILE_UNITS='degrees',",
                    "IRAMP=0,",
                    "SCALAR_POSITIVITY_CONTROL=T,",
                    "NOFLUX_BOT_CONDITION=T,",
                    "STARTUP_FILE='initial.nc',",
                    f"START_DATE='{(START - timedelta(days=day)).isoformat()}',",
                    f"END_DATE='{self.times[-1].isoformat()}',",
                    "HEATING_LONGWAVE_PERCTAGE=0.55,",
                    "HEATING_LONGWAVE_LENGTHSCALE=3.6,",
                    "HEATING_SHORTWAVE_LENGTHSCALE=0.35,",
                ]
            )
        )
        acct = self.d / f"acct{day}.txt"
        acct.write_text(f"{day}|COMPLETED|0:0|")
        files = {r["path"]: r["sha256"] for r in self.force.values()}
        files["initial.nc"] = str(day)
        attempt = self.js(
            f"att{day}.json",
            dict(
                site="example-cpu",
                account="example",
                mpi_ranks=4,
                files=files,
                run_namelist_sha256=sha(nml),
                executable_sha256="a" * 64,
                job_sha256="b" * 64,
            ),
        )
        checkpoint = self.d / f"checkpoint{day}.nc"
        checkpoint.write_bytes(f"native restart fixture for {day}".encode())
        identifier = f"trial_{day}"
        restart = dict(
            schema="fvcom_lake_restart_v1",
            status="verified",
            attempt=identifier,
            job_id=str(day),
            utc=START.isoformat(),
            native_path=str(checkpoint),
            path=str(checkpoint),
            sha256=sha(checkpoint),
        )
        native = self.js(
            f"native{day}.json",
            dict(
                schema="fvcom_lake_native_audit_v1",
                status="verified",
                attempt_sha256=attempt["sha256"],
                executable_sha256="a" * 64,
                job_id=str(day),
                attempt=identifier,
                stage=f"spinup_{day}d",
                first_utc=(START - timedelta(days=day)).isoformat(),
                last_utc=self.times[-1].isoformat(),
                startup_sha256=str(day),
                output_files={
                    path.name: dict(sha256=sha(path)),
                    checkpoint.name: dict(
                        sha256=sha(checkpoint), first=START.isoformat(), last=START.isoformat(), count=1
                    ),
                },
                restart_receipts=dict(comparison=restart),
            ),
        )
        job = dict(
            job_id=str(day),
            accounting=self.ref(acct),
            submission=self.js(
                f"sub{day}.json", dict(account="example", job_id=str(day), script_sha256="b" * 64)
            ),
        )
        return dict(
            job=job,
            attempt=attempt,
            namelist=self.ref(nml),
            forcing=self.force,
            outputs=[self.ref(path)],
            freeze=self.freeze,
            native_audit=native,
        )

    def test_spinup_and_eligibility(self):
        result = spinup(self.d, dict(stage_plan=self.plan, members=self.members), self.geom)
        self.assertEqual(result["selection_days"], 6)
        self.assertEqual(result["selected_restart"]["job_id"], "6")
        self.assertFalse(result["pairs"]["3-6"]["passed"])
        self.assertTrue(result["pairs"]["6-9"]["passed"])
        spec = {"members": {"9": self.members["9"]}, "stage_plan": self.plan}
        self.assertEqual(spinup(self.d, spec, self.geom)["status"], "advisory_unresolved")
        self.assertEqual(
            spinup(self.d, {"members": {}, "stage_plan": self.plan}, self.geom)["status"], "blocked"
        )

    def test_reject_modified_restart(self):
        (self.d / "checkpoint6.nc").write_bytes(b"changed")
        result = spinup(self.d, dict(stage_plan=self.plan, members=self.members), self.geom)
        self.assertEqual(result["status"], "advisory_unresolved")
        self.assertIn("SHA-256 differs", result["rejected"][6])
        self.assertEqual(result["selected_restart"]["job_id"], "9")

    def screened_spec(self, cold=None):
        table = self.d/"freshwater.csv"
        table.write_text("sea_pressure_dbar,air_free_temperature_C,air_saturated_temperature_C\n-1,0,-.001\n100,-.05,-.051\n")
        policy = self.js("thermal.json",dict(freezing_lookup_sha256=sha(table),cold_conservative_margin_C=.01,cold_substantial_margin_C=.10))
        plan = json.loads((self.d/self.plan["path"]).read_text())
        plan.update(profile="lake_superior_summer",freshwater_thermal_screen=dict(policy_sha256=policy["sha256"],freezing_lookup_sha256=sha(table)))
        rec = self.gen(9, 0, full_window=True, cold=cold)
        spec = dict(stage_plan=self.js("thermal_plan.json",plan), thermal_policy=policy,
                    freezing_lookup=self.ref(table), members={"9":rec}, geometry=self.geometry)
        m = member(self.d,rec,self.geom,START-timedelta(days=9),self.times[-1])
        receipt = thermal_measurement(self.d,spec,rec,m,self.geom,thermal_settings(self.d,spec))
        return spec,receipt

    def test_wet_screen_and_preregistration(self):
        spec, receipt = self.screened_spec()
        rec = spec["members"]["9"]
        self.assertEqual(spinup(self.d,spec,self.geom)["status"],"blocked")
        rec["thermal_manifest"] = self.js("thermal_run.json",receipt)
        self.assertEqual(spinup(self.d,spec,self.geom)["selection_days"],9)
        for change in (dict(job_id="unexpected"),dict(native_audit_sha256="0"*64),dict(records=receipt["records"][1:]),
                       dict(policy_sha256="0"*64),dict(last_utc=START.isoformat()),dict(status="adjudication_required")):
            rec["thermal_manifest"] = self.js("thermal_run.json",{**receipt,**change})
            self.assertIsNone(spinup(self.d,spec,self.geom)["selected_restart"])
        self.js("request.json",dict(profile="lake_superior_summer"))
        with self.assertRaisesRegex(ValueError,"preregistered"):
            spinup(self.d,dict(stage_plan=self.plan,members=self.members),self.geom)
        with self.assertRaisesRegex(ValueError,"changed preregistered"):
            spinup(self.d,{**spec,"thermal_policy":self.freeze},self.geom)

    def test_freezing_even_before_comparison_blocks_restart(self):
        for cold in (-3.,-.06):
            spec,receipt=self.screened_spec(cold=cold)
            self.assertEqual(receipt["status"],"adjudication_required")
            receipt["status"]="qualified"
            spec["members"]["9"]["thermal_manifest"]=self.js("falsely_qualified.json",receipt)
            self.assertIsNone(spinup(self.d,spec,self.geom)["selected_restart"])

    def test_portable_wet_screen_commands(self):
        spec, _ = self.screened_spec()
        rec = spec["members"]["9"]
        folder = self.d/"screen_run"
        folder.mkdir()
        for ref, name in ((rec["namelist"],"test_run.nml"),(rec["attempt"],"attempt_manifest.json"),
                          (rec["outputs"][0],"output9.nc")):
            shutil.copyfile(self.d/ref["path"],folder/name)
        inventory = {k: spec[k]["path"] for k in ("stage_plan","thermal_policy","freezing_lookup")}
        inventory["geometry"] = rec["outputs"][0]["path"]
        paths = dict(attempt_directory=str(folder), history_glob="output*.nc", freeze=rec["freeze"]["path"],
                     native_audit=rec["native_audit"]["path"],accounting=rec["job"]["accounting"]["path"],
                     submission=rec["job"]["submission"]["path"],forcing={k:f["path"] for k,f in rec["forcing"].items()})
        scripts = Path(__file__).resolve().parent
        for mode in ("thermal-screen","spinup"):
            self.js("inventory.json",{**inventory,**({"member":paths} if mode=="thermal-screen" else
                {"members":{"9":{**paths,"thermal_manifest":"thermal-screen.json"}}})})
            for name,flags in (("prepare_lake_physics.py",["--inventory","inventory.json","--output","bound.json"]),
                               ("lake_physics.py",["--spec","bound.json","--output",mode+".json"])):
                subprocess.run([sys.executable,str(scripts/name),mode,*flags],cwd=self.d,check=True)
        result=json.loads((self.d/"spinup.json").read_text())
        self.assertEqual(result["selected_restart"]["job_id"],"9")
        self.assertEqual(result["accepted"]["9"]["freshwater_thermal_screen"]["report_count"],49)

    def test_portable_commands(self):
        scripts = Path(__file__).resolve().parent

        def command(program, *args):
            subprocess.run([sys.executable, str(scripts / program), *args], cwd=self.d, check=True)

        command(
            "prepare_lake_physics.py", "geometry", "--source", self.geometry["path"], "--output", "geom.nc"
        )
        paths = {}
        for day, obj in self.members.items():
            folder = self.d / f"run{day}"
            folder.mkdir()
            shutil.copyfile(self.d / obj["namelist"]["path"], folder / "test_run.nml")
            shutil.copyfile(self.d / obj["attempt"]["path"], folder / "attempt_manifest.json")
            hist = obj["outputs"][0]["path"]
            shutil.copyfile(self.d / hist, folder / hist)
            paths[day] = dict(
                attempt_directory=str(folder),
                history_glob="output*.nc",
                freeze=self.freeze["path"],
                native_audit=obj["native_audit"]["path"],
                accounting=obj["job"]["accounting"]["path"],
                submission=obj["job"]["submission"]["path"],
                forcing={k: v["path"] for k, v in self.force.items()},
            )
        for mode, inventory in (
            ("spinup", dict(members=paths, stage_plan=self.plan["path"])),
            (
                "budget",
                dict(
                    member=paths["9"],
                    start_utc=START.isoformat(),
                    end_utc=self.times[1].isoformat(),
                    surface_budget=self.surface_csv.name,
                    surface_manifest=self.surf["path"],
                    river_manifest=self.riv["path"],
                ),
            ),
        ):
            self.js("inputs.json", dict(geometry="geom.nc", **inventory))
            command("prepare_lake_physics.py", mode, "--inventory", "inputs.json", "--output", "spec.json")
            command("lake_physics.py", mode, "--spec", "spec.json", "--output", "result.json")
            result = json.loads((self.d / "result.json").read_text())
            if mode == "spinup":
                self.assertEqual(result["selected_restart"]["job_id"], "6")
            else:
                self.assertEqual(result["totals"]["water_residual_m3"], 0)

    def test_equal_in_out_heat_budget(self):
        r = budget(
            self.d,
            dict(
                start_utc=START.isoformat(),
                end_utc=self.times[1].isoformat(),
                member=self.members["9"],
                surface_budget=self.ref(self.surface_csv),
                surface_manifest=self.surf,
                river_manifest=self.riv,
            ),
            self.geom,
        )["totals"]
        self.assertEqual(r["water_residual_m3"], 0)
        self.assertEqual(r["heat_residual_j"], 0)
        self.assertEqual(r["river_in_heat_j"], -r["river_donor_out_heat_j"])
        self.assertGreater(r["river_in_heat_j"], 0)


if __name__ == "__main__":
    unittest.main()
