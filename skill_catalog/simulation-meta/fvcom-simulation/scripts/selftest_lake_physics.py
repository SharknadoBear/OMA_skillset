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

from lake_physics import setup, sha, spinup
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
                self.clock(f, [START + timedelta(hours=i) for i in range(73)])
                f.createDimension("node", 3)
                if name == "river":
                    f.createDimension("rivers", 2)
                    f.createDimension("namelen", 2)
                    text = f.createVariable("river_names", "S1", ("rivers", "namelen"))
                    text[:] = np.asarray([["r", "1"], ["r", "2"]], dtype="S1")
                    v = f.createVariable("river_flux", "f8", ("time", "rivers"))
                    v[:] = [[1, -1]] * 73
                    v = f.createVariable("river_temp", "f8", ("time", "rivers"))
                    v[:] = [[10, 0]] * 73
                else:
                    for key in ("precip", "evap") if name == "pe" else ("net_heat_flux", "short_wave"):
                        f.createVariable(key, "f8", ("time", "node"))[:] = 0.0
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
                count=73,
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

    def gen(self, day, offset):
        path = self.d / f"output{day}.nc"
        with nc.Dataset(path, "w") as f:
            f.CoordinateSystem = "GeoReferenced"
            self.clock(f, self.times)
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
                np.array([16, 12, 8, 4])[None, :, None] + offset, (len(self.times), 4, 3)
            )
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
