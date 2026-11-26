"""Make a SHA-256-bound spinup or budget audit specification from actual run-control files."""

import argparse
import json
from pathlib import Path

import netCDF4 as nc

from lake_physics import clocks, policy, sha, thermal_settings, utc


def ref(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": sha(path)}


def geom(source, output):
    with nc.Dataset(source) as ds, nc.Dataset(output, "w") as dst:
        for key in ("node", "nele", "siglev", "three"):
            dst.createDimension(key, len(ds.dimensions[key]))
        for key in ("nv", "h", "lon", "lat", "siglev", "art1"):
            v = ds[key]
            w = dst.createVariable(key, v.dtype, v.dimensions, zlib=True, complevel=2)
            w.setncatts({k: v.getncattr(k) for k in v.ncattrs() if k != "_FillValue"})
            w[:] = v[:]
        dst.source_sha256 = sha(source)
        dst.CoordinateSystem = ds.CoordinateSystem


def make_member(inventory, start, end):
    root = Path(inventory["attempt_directory"])
    attempt = root / "attempt_manifest.json"
    payload = json.loads(attempt.read_text())
    namelists = [f for f in root.glob("*_run.nml") if sha(f) == payload["run_namelist_sha256"]]
    if len(namelists) != 1:
        raise ValueError("Expected exactly one runtime namelist matching the producing attempt")
    prefix = namelists[0].name.removesuffix("_run.nml")
    grid_files = sorted(root.glob(inventory.get("history_glob", prefix + "_[0-9][0-9][0-9][0-9].nc")))

    def relevant(path):
        with nc.Dataset(path) as f:
            return any(start <= t <= end for t in clocks(f))

    grid_files = [f for f in grid_files if relevant(f)]
    if not grid_files:
        raise ValueError(f"No returned gridded output in {root}")
    for key in ("submission", "accounting", "freeze", "native_audit"):
        if key not in inventory:
            raise ValueError(f"Missing provenance input {key}")
    sub = json.loads(Path(inventory["submission"]).read_text())
    return dict(
        attempt=ref(attempt),
        namelist=ref(namelists[0]),
        freeze=ref(inventory["freeze"]),
        native_audit=ref(inventory["native_audit"]),
        job=dict(
            job_id=sub["job_id"],
            accounting=ref(inventory["accounting"]),
            submission=ref(inventory["submission"]),
        ),
        outputs=[ref(f) for f in grid_files],
        forcing={key: ref(path) for key, path in inventory["forcing"].items()},
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    s = sub.add_parser("geometry")
    s.add_argument("--source", required=True, type=Path)
    s.add_argument("--output", required=True, type=Path)
    for action in ("spinup", "budget", "thermal-screen"):
        s = sub.add_parser(action)
        s.add_argument("--inventory", type=Path, required=True)
        s.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    if a.action == "geometry":
        if a.output.exists():
            raise ValueError("Supply a fresh geometry output")
        geom(a.source, a.output)
    else:
        d = json.loads(a.inventory.read_text())
        result = dict(geometry=ref(d["geometry"]))
        if a.action in ("spinup", "thermal-screen"):
            result["stage_plan"] = ref(d["stage_plan"])
            plan_path = Path(d["stage_plan"])
            plan = json.loads(plan_path.read_text())
            adjacent = plan_path.with_name("request.json")
            if "project_request" in d or adjacent.is_file():
                result["project_request"] = ref(d.get("project_request", adjacent))
            for key, plan_key in (("thermal_policy", "policy"), ("freezing_lookup", "freezing_lookup")):
                if key in d:
                    result[key] = ref(d[key])
                elif plan.get("freshwater_thermal_screen", {}).get(plan_key):
                    result[key] = ref(plan_path.parent / plan["freshwater_thermal_screen"][plan_key])
            screened = thermal_settings(Path("."), result) is not None
        if a.action == "spinup":
            start, end, _, _ = policy(Path("."), result)
            result["members"] = {}
            result["excluded"] = {}
            for key, value in d["members"].items():
                try:
                    if screened:
                        native = json.loads(Path(value["native_audit"]).read_text())
                        full = make_member(value, utc(native["first_utc"]), utc(native["last_utc"]))
                        full["thermal_manifest"] = ref(value["thermal_manifest"])
                        result["members"][key] = full
                    else:
                        result["members"][key] = make_member(value, start, end)
                except (KeyError, OSError, ValueError) as err:
                    result["excluded"][key] = str(err)
        elif a.action == "thermal-screen":
            if not screened:
                raise ValueError("Preregister the requested thermal screen before preparing the native member")
            native = json.loads(Path(d["member"]["native_audit"]).read_text())
            result["member"] = make_member(d["member"], utc(native["first_utc"]), utc(native["last_utc"]))
        else:
            result["member"] = make_member(d["member"], utc(d["start_utc"]), utc(d["end_utc"]))
            for key in ("surface_manifest", "river_manifest", "surface_budget"):
                result[key] = ref(d[key])
            for key in ("start_utc", "end_utc"):
                result[key] = d[key]
        a.output.write_text(json.dumps(result, indent=2) + "\n")
